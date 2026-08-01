from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()

# Input (Sepal Length)
X = iris.data[:, 0].reshape(-1, 1)

# Output (Sepal Width)
y = iris.data[:, 1]

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Predict values
y_pred = model.predict(X)

# Plot Actual vs Predicted
plt.figure(figsize=(10,6))
plt.scatter(X, y, color='green', marker='x', label='Actual')
plt.plot(X, y_pred, color='red', linewidth=3, label='Predicted')

# Labels and title
plt.title("Linear Regression: Sepal Width vs Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.legend()

# Show graph
plt.show()

# Print model performance
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)
print("R² Score:", model.score(X, y))
