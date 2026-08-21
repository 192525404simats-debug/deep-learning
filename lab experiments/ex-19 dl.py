import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score

# Create circular dataset
X, y = make_circles(
    n_samples=200,
    noise=0.1,
    factor=0.5,
    random_state=1
)

# Create Neural Network
model = MLPClassifier(
    hidden_layer_sizes=(3, 3),
    activation='identity',
    learning_rate_init=0.03,
    max_iter=1000,
    random_state=1
)

# Train the model
model.fit(X, y)

# Prediction
y_pred = model.predict(X)

# Display accuracy
print("Accuracy:", accuracy_score(y, y_pred))

# Plot circular data
plt.figure(figsize=(10, 7))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap="viridis",
    s=60
)

plt.title("Neural Network - Circular Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()
