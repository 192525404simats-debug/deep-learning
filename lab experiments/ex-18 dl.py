import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Create dataset
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=1
)

# Create Neural Network model
model = MLPClassifier(
    hidden_layer_sizes=(3, 3),
    activation='identity',
    learning_rate_init=0.03,
    max_iter=1000,
    random_state=1
)

# Train the model
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Display accuracy
print("Accuracy:", accuracy_score(y, y_pred))

# Plot the two classes
plt.figure(figsize=(10, 7))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap="viridis",
    s=60
)

plt.title("Neural Network - Two Class")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()
