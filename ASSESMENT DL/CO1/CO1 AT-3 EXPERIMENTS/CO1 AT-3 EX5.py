import numpy as np
import matplotlib.pyplot as plt

# Input values
inputs = np.array([1, 2, 3])

# Weights
weights = np.array([0.5, -0.3, 0.8])

# Bias
bias = 0.2

# Calculate weighted sum
z = np.dot(inputs, weights) + bias

# Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ReLU Activation Function
def relu(x):
    return np.maximum(0, x)

# Outputs
sigmoid_output = sigmoid(z)
relu_output = relu(z)

# Print Results
print("Inputs:", inputs)
print("Weights:", weights)
print("Bias:", bias)
print("Weighted Sum:", round(z, 2))
print("Sigmoid Output:", round(sigmoid_output, 4))
print("ReLU Output:", round(relu_output, 4))

# Plot Comparison
activations = ["Sigmoid", "ReLU"]
outputs = [sigmoid_output, relu_output]

plt.bar(activations, outputs)
plt.title("Comparison of Activation Functions")
plt.ylabel("Output")
plt.ylim(0, max(outputs) + 1)
plt.show()
