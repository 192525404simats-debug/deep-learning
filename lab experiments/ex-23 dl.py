import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# Create spiral dataset
n = 200

t = np.linspace(0, 4 * np.pi, n)
r = np.linspace(0.1, 1, n)

# First spiral
X1 = np.c_[r * np.cos(t), r * np.sin(t)]

# Second spiral
X2 = np.c_[r * np.cos(t + np.pi), r * np.sin(t + np.pi)]

# Combine both spirals
X = np.vstack((X1, X2))

# Create labels
y = np.array([0] * n + [1] * n)

# Create Neural Network with Sigmoid activation
model = MLPClassifier(
    hidden_layer_sizes=(3, 3, 3),
    activation='logistic',
    learning_rate_init=0.1,
    max_iter=3000,
    random_state=1
)

# Train the model
model.fit(X, y)

# Display accuracy
print("Accuracy:", model.score(X, y))

# Plot spiral data
plt.figure(figsize=(8, 6))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap="viridis",
    s=35
)

plt.title("Spiral Data - Sigmoid")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()
