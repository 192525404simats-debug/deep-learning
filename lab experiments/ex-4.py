import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate sample data
np.random.seed(0)
X = np.sort(np.random.rand(20))
y = np.cos(1.5 * np.pi * X) + np.random.randn(20) * 0.1

X = X[:, np.newaxis]
X_test = np.linspace(0, 1, 100)[:, np.newaxis]

degrees = [1, 4, 15]

plt.figure(figsize=(15, 5))

for i, degree in enumerate(degrees):
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("linear", LinearRegression())
    ])

    model.fit(X, y)

    y_pred = model.predict(X_test)
    train_pred = model.predict(X)

    mse = mean_squared_error(y, train_pred)

    plt.subplot(1, 3, i + 1)
    plt.scatter(X, y, color="magenta", s=15, label="Samples")
    plt.plot(X_test, y_pred, color="blue", label="Model")
    plt.plot(X_test, np.cos(1.5 * np.pi * X_test),
             color="orange", label="True function")

    plt.title(f"Degree {degree}\nMSE = {mse:.2e}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

plt.tight_layout()
plt.show()
