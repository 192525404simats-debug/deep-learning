import numpy as np

# Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Inputs
x = np.array([0.5, 0.8])

# Input to Hidden Layer Weights (2x2)
W1 = np.array([[0.1, 0.4],
               [0.3, 0.2]])

# Hidden Layer Bias
b1 = np.array([0.2, 0.1])

# Hidden to Output Layer Weights (2x1)
W2 = np.array([[0.7],
               [0.5]])

# Output Layer Bias
b2 = np.array([0.3])

# -------- Forward Propagation --------

# Hidden Layer
hidden_input = np.dot(x, W1) + b1
hidden_output = sigmoid(hidden_input)

# Output Layer
final_input = np.dot(hidden_output, W2) + b2
final_output = sigmoid(final_input)

print("Hidden Layer Input:")
print(hidden_input)

print("\nHidden Layer Output:")
print(hidden_output)

print("\nOutput Layer Input:")
print(final_input)

print("\nFinal Output:")
print(final_output)
