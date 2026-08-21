import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier

# Create multi-class dataset
X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_classes=3,
    n_clusters_per_class=1,
    n_redundant=0,
    random_state=1
)

# Create Neural Network with Sigmoid activation
model = MLPClassifier(
    hidden_layer_sizes=(3, 3, 3),
    activation='logistic',
    learning_rate_init=0.1,
    max_iter=2000,
    random_state=1
)

# Train the model
model.fit(X, y)

# Display accuracy
print("Accuracy:", model.score(X, y))

# Plot the data
plt.figure(figsize=(8, 6))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap="viridis",
    s=50
)

plt.title("Multi Class Data - Sigmoid")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()
