import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create dataset
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    random_state=42
)

# Create Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Calculate accuracy
print("Accuracy:", accuracy_score(y, y_pred))

# Plot data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", s=50)

# Get model coefficients
coef = model.coef_[0]
intercept = model.intercept_[0]

# Create decision boundary
x = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
line = -(coef[0] * x + intercept) / coef[1]

# Plot decision boundary
plt.plot(x, line)

# Labels and title
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Linear Separability")

plt.show()
