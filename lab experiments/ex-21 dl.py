import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.neural_network import MLPClassifier

# Create circular dataset
X, y = make_circles(
    n_samples=200,
    noise=0.1,
    factor=0.5,
    random_state=1
)

# Create Neural Network with ReLU activation
model = MLPClassifier(
    hidden_layer_sizes=(2, 2, 2),
    activation='relu',
    learning_rate_init=0.1,
    max_iter=2000,
    random_state=1
)

# Train the model
model.fit(X, y)

# Display accuracy
print("Accuracy:", model.score(X, y))

# Plot circular data
plt.figure(figsize=(8, 6))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap="viridis",
    s=50
)

plt.title("Circular Data - ReLU")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()
