from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# -------------------------------
# Linearly Separable Dataset
# -------------------------------
X1, y1 = make_classification(
    n_samples=100,
    n_features=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

model1 = Perceptron(max_iter=1000, random_state=42)
model1.fit(X1, y1)

pred1 = model1.predict(X1)

print("Linearly Separable Dataset")
print("Accuracy:", accuracy_score(y1, pred1))

# -------------------------------
# Non-Linearly Separable Dataset (XOR)
# -------------------------------
X2 = [[0,0],[0,1],[1,0],[1,1]]
y2 = [0,1,1,0]

model2 = Perceptron(max_iter=1000, random_state=42)
model2.fit(X2, y2)

pred2 = model2.predict(X2)

print("\nNon-Linearly Separable Dataset (XOR)")
print("Accuracy:", accuracy_score(y2, pred2))

# Plot Linearly Separable Dataset
plt.scatter(X1[:,0], X1[:,1], c=y1, cmap='bwr')
plt.title("Linearly Separable Dataset")
plt.show()
