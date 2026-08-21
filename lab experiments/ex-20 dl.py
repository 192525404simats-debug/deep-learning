import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Create multi-class dataset
X, y = make_classification(
    n_samples=150,
    n_features=2,
    n_classes=3,
    n_clusters_per_class=1,
    n_redundant=0,
    random_state=1
)

# Create Neural Network model
model = MLPClassifier(
    hidden_layer_sizes=(2, 2),
    activation='identity',
    learning_rate_init=0.01,
    max_iter=1000,
    random_state=1
)

# Train the model
model.fit(X, y)

# Prediction
y_pred = model.predict(X)

# Display accuracy
print("Accuracy:", accuracy_score(y, y_pred))

# Plot the three classes
plt.figure(figsize=(10, 7))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap="viridis",
    s=60
)

plt.title("Neural Network - Multi Class")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()
