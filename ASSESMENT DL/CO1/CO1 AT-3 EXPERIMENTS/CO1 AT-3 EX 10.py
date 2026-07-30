import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import pdist

# Dimensions to test
dimensions = [2, 5, 10, 20, 50, 100]

avg_distances = []
accuracies = []

for dim in dimensions:

    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=500,
        n_features=dim,
        n_informative=min(5, dim),
        n_redundant=0,
        random_state=42
    )

    # Average pairwise distance
    avg_distance = np.mean(pdist(X))
    avg_distances.append(avg_distance)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train KNN classifier
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    print(f"Dimension = {dim}")
    print(f"Average Distance = {avg_distance:.2f}")
    print(f"Accuracy = {accuracy:.4f}")
    print()

# Plot Average Distance
plt.figure(figsize=(8,5))
plt.plot(dimensions, avg_distances, marker='o')
plt.title("Average Distance vs Dimensions")
plt.xlabel("Dimensions")
plt.ylabel("Average Distance")
plt.grid(True)
plt.show()

# Plot Accuracy
plt.figure(figsize=(8,5))
plt.plot(dimensions, accuracies, marker='o')
plt.title("KNN Accuracy vs Dimensions")
plt.xlabel("Dimensions")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()
