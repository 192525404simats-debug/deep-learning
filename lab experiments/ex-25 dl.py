from sklearn.datasets import make_circles
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

X, y = make_circles(n_samples=200, noise=0.1,
                    factor=0.5, random_state=1)

model = MLPClassifier(hidden_layer_sizes=(2,2),
                      activation='tanh',
                      learning_rate_init=0.1,
                      max_iter=2000,
                      random_state=1)

model.fit(X, y)
print("Accuracy:", model.score(X, y))

plt.scatter(X[:,0], X[:,1], c=y)
plt.title("Circular Data - Tanh")
plt.show()
