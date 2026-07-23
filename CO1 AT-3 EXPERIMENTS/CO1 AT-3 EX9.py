import numpy as np
import matplotlib.pyplot as plt

# Dataset
X = np.array([1,2,3,4,5,6,7,8,9,10], dtype=float)
Y = np.array([2,4,6,8,10,12,14,16,18,20], dtype=float)

# Parameters
learning_rate = 0.01
epochs = 100
batch_size = 2      # Change to 1, 2, 5, or 10

# Initialize weights
m = 0
b = 0

cost_history = []

n = len(X)

for epoch in range(epochs):

    # Shuffle data every epoch
    indices = np.arange(n)
    np.random.shuffle(indices)

    X_shuffled = X[indices]
    Y_shuffled = Y[indices]

    # Process mini-batches
    for i in range(0, n, batch_size):

        X_batch = X_shuffled[i:i+batch_size]
        Y_batch = Y_shuffled[i:i+batch_size]

        # Prediction
        Y_pred = m * X_batch + b

        # Gradients
        dm = (-2/len(X_batch)) * np.sum(X_batch * (Y_batch - Y_pred))
        db = (-2/len(X_batch)) * np.sum(Y_batch - Y_pred)

        # Update parameters
        m = m - learning_rate * dm
        b = b - learning_rate * db

    # Calculate cost after each epoch
    predictions = m * X + b
    cost = np.mean((Y - predictions) ** 2)
    cost_history.append(cost)

print("Final Slope (m):", round(m,4))
print("Final Intercept (b):", round(b,4))

# Plot Cost
plt.plot(cost_history)
plt.title("Mini-Batch Gradient Descent")
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.grid(True)
plt.show()
